"""
Pandera Schema Definitions for Synthetic Data.

Validates DataFrame structure, data types, and value constraints.
"""

from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
from pandera.typing import Series

from ..config import (
    Brand,
    DataSplit,
    SpecialtyEnum,
    PracticeTypeEnum,
    RegionEnum,
    InsuranceTypeEnum,
    EngagementTypeEnum,
)


# =============================================================================
# ENUM VALUE LISTS (for validation)
# =============================================================================

BRAND_VALUES = [b.value for b in Brand]
DATA_SPLIT_VALUES = [s.value for s in DataSplit]
SPECIALTY_VALUES = [s.value for s in SpecialtyEnum]
PRACTICE_TYPE_VALUES = [p.value for p in PracticeTypeEnum]
REGION_VALUES = [r.value for r in RegionEnum]
INSURANCE_TYPE_VALUES = [i.value for i in InsuranceTypeEnum]
ENGAGEMENT_TYPE_VALUES = [e.value for e in EngagementTypeEnum]

# Treatment event types (from Supabase schema)
TREATMENT_EVENT_TYPES = [
    "diagnosis", "prescription", "lab_test",
    "procedure", "consultation", "hospitalization"
]

# Prediction types (from Supabase schema)
PREDICTION_TYPES = [
    "trigger", "propensity", "risk", "churn", "next_best_action"
]

# Trigger types
TRIGGER_TYPES = [
    "prescription_opportunity", "adherence_risk", "churn_prevention",
    "cross_sell", "engagement_gap", "competitive_threat",
    "treatment_switch", "reactivation"
]

# Delivery channels
DELIVERY_CHANNELS = ["email", "call", "in_person", "portal"]

# Status enums
DELIVERY_STATUS_VALUES = ["pending", "sent", "delivered", "failed"]
ACCEPTANCE_STATUS_VALUES = ["pending", "accepted", "rejected", "expired"]


# =============================================================================
# HCP PROFILE SCHEMA
# =============================================================================

HCPProfileSchema = DataFrameSchema(
    columns={
        "hcp_id": Column(
            str,
            Check.str_matches(r"^hcp_\d+$"),
            unique=True,
            description="Unique HCP identifier (format: hcp_XXXXX)",
        ),
        "npi": Column(
            str,
            Check.str_matches(r"^\d{10}$"),
            nullable=False,
            description="10-digit NPI number",
        ),
        "specialty": Column(
            str,
            Check.isin(SPECIALTY_VALUES),
            nullable=False,
            description="HCP specialty",
        ),
        "practice_type": Column(
            str,
            Check.isin(PRACTICE_TYPE_VALUES),
            nullable=False,
            description="Practice type (academic/community/private)",
        ),
        "geographic_region": Column(
            str,
            Check.isin(REGION_VALUES),
            nullable=False,
            description="Geographic region",
        ),
        "years_experience": Column(
            int,
            Check.in_range(0, 50),
            nullable=False,
            description="Years of experience",
        ),
        "academic_hcp": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description="Academic affiliation flag",
        ),
        "total_patient_volume": Column(
            int,
            Check.in_range(10, 1000),
            nullable=False,
            description="Total patient volume",
        ),
        "brand": Column(
            str,
            Check.isin(BRAND_VALUES),
            nullable=False,
            description="Associated brand",
        ),
    },
    strict=False,  # Allow extra columns
    coerce=True,   # Coerce types when possible
    name="HCPProfileSchema",
    description="Schema for HCP profile records",
)


# =============================================================================
# PATIENT JOURNEY SCHEMA
# =============================================================================

PatientJourneySchema = DataFrameSchema(
    columns={
        "patient_journey_id": Column(
            str,
            Check.str_matches(r"^patient_\d+$"),
            unique=True,
            description="Unique journey identifier",
        ),
        "patient_id": Column(
            str,
            Check.str_matches(r"^pt_\d+$"),
            nullable=False,
            description="Patient identifier",
        ),
        "hcp_id": Column(
            str,
            Check.str_matches(r"^hcp_\d+$"),
            nullable=False,
            description="Treating HCP identifier",
        ),
        "brand": Column(
            str,
            Check.isin(BRAND_VALUES),
            nullable=False,
            description="Treatment brand",
        ),
        "journey_start_date": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Journey start date (YYYY-MM-DD)",
        ),
        "data_split": Column(
            str,
            Check.isin(DATA_SPLIT_VALUES),
            nullable=False,
            description="ML data split assignment",
        ),
        "disease_severity": Column(
            float,
            Check.in_range(0.0, 10.0),
            nullable=False,
            description="Disease severity score (0-10)",
        ),
        "academic_hcp": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description="Academic HCP flag (confounder)",
        ),
        "engagement_score": Column(
            float,
            Check.in_range(0.0, 10.0),
            nullable=False,
            description="Engagement score (treatment variable)",
        ),
        "treatment_initiated": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
            description="Treatment initiation (outcome variable)",
        ),
        "days_to_treatment": Column(
            object,  # Can be int or None
            nullable=True,
            description="Days to treatment initiation",
        ),
        "geographic_region": Column(
            str,
            Check.isin(REGION_VALUES),
            nullable=False,
            description="Geographic region",
        ),
        "insurance_type": Column(
            str,
            Check.isin(INSURANCE_TYPE_VALUES),
            nullable=False,
            description="Insurance type",
        ),
        "age_at_diagnosis": Column(
            int,
            Check.in_range(0, 120),
            nullable=False,
            description="Age at diagnosis",
        ),
    },
    strict=False,
    coerce=True,
    name="PatientJourneySchema",
    description="Schema for patient journey records with embedded causal structure",
)


# =============================================================================
# TREATMENT EVENT SCHEMA
# =============================================================================

TreatmentEventSchema = DataFrameSchema(
    columns={
        "treatment_event_id": Column(
            str,
            Check.str_matches(r"^trx_\d+$"),
            unique=True,
            description="Unique treatment event identifier",
        ),
        "patient_journey_id": Column(
            str,
            Check.str_matches(r"^patient_\d+$"),
            nullable=False,
            description="Parent journey identifier",
        ),
        "patient_id": Column(
            str,
            Check.str_matches(r"^pt_\d+$"),
            nullable=False,
            description="Patient identifier",
        ),
        "hcp_id": Column(
            str,
            Check.str_matches(r"^hcp_\d+$"),
            nullable=False,
            description="Treating HCP identifier",
        ),
        "brand": Column(
            str,
            Check.isin(BRAND_VALUES),
            nullable=False,
            description="Treatment brand",
        ),
        "event_date": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Event date (YYYY-MM-DD)",
        ),
        "event_type": Column(
            str,
            Check.isin(TREATMENT_EVENT_TYPES),
            nullable=False,
            description="Type of treatment event",
        ),
        "duration_days": Column(
            int,
            Check.in_range(1, 365),
            nullable=False,
            description="Duration in days",
        ),
        "refill_number": Column(
            int,
            Check.in_range(0, 50),
            nullable=False,
            description="Refill sequence number",
        ),
        "adherence_score": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Adherence score (0-1)",
        ),
        "efficacy_score": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Efficacy score (0-1)",
        ),
        "data_split": Column(
            str,
            Check.isin(DATA_SPLIT_VALUES),
            nullable=False,
            description="ML data split assignment",
        ),
    },
    strict=False,
    coerce=True,
    name="TreatmentEventSchema",
    description="Schema for treatment event records",
)


# =============================================================================
# ML PREDICTION SCHEMA
# =============================================================================

MLPredictionSchema = DataFrameSchema(
    columns={
        "prediction_id": Column(
            str,
            Check.str_matches(r"^pred_\d+$"),
            unique=True,
            description="Unique prediction identifier",
        ),
        "patient_journey_id": Column(
            str,
            Check.str_matches(r"^patient_\d+$"),
            nullable=False,
            description="Parent journey identifier",
        ),
        "patient_id": Column(
            str,
            Check.str_matches(r"^pt_\d+$"),
            nullable=False,
            description="Patient identifier",
        ),
        "hcp_id": Column(
            str,
            Check.str_matches(r"^hcp_\d+$"),
            nullable=False,
            description="Associated HCP identifier",
        ),
        "brand": Column(
            str,
            Check.isin(BRAND_VALUES),
            nullable=False,
            description="Brand context",
        ),
        "prediction_type": Column(
            str,
            Check.isin(PREDICTION_TYPES),
            nullable=False,
            description="Type of ML prediction",
        ),
        "prediction_value": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Prediction value (probability)",
        ),
        "confidence_score": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Model confidence score",
        ),
        "uncertainty": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Prediction uncertainty",
        ),
        "model_version": Column(
            str,
            Check.str_matches(r"^v\d+\.\d+$"),
            nullable=False,
            description="Model version string",
        ),
        "prediction_timestamp": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Prediction timestamp (YYYY-MM-DD)",
        ),
        "data_split": Column(
            str,
            Check.isin(DATA_SPLIT_VALUES),
            nullable=False,
            description="ML data split assignment",
        ),
    },
    strict=False,
    coerce=True,
    name="MLPredictionSchema",
    description="Schema for ML prediction records",
)


# =============================================================================
# TRIGGER SCHEMA
# =============================================================================

TriggerSchema = DataFrameSchema(
    columns={
        "trigger_id": Column(
            str,
            Check.str_matches(r"^trig_\d+$"),
            unique=True,
            description="Unique trigger identifier",
        ),
        "patient_id": Column(
            str,
            Check.str_matches(r"^pt_\d+$"),
            nullable=False,
            description="Target patient identifier",
        ),
        "hcp_id": Column(
            str,
            Check.str_matches(r"^hcp_\d+$"),
            nullable=False,
            description="Target HCP identifier",
        ),
        "trigger_timestamp": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Trigger timestamp (YYYY-MM-DD)",
        ),
        "trigger_type": Column(
            str,
            Check.isin(TRIGGER_TYPES),
            nullable=False,
            description="Type of trigger",
        ),
        "priority": Column(
            int,
            Check.in_range(1, 5),
            nullable=False,
            description="Priority level (1=highest, 5=lowest)",
        ),
        "confidence_score": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Trigger confidence score",
        ),
        "lead_time_days": Column(
            int,
            Check.in_range(0, 365),
            nullable=False,
            description="Lead time in days",
        ),
        "expiration_date": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Trigger expiration date",
        ),
        "delivery_channel": Column(
            str,
            Check.isin(DELIVERY_CHANNELS),
            nullable=False,
            description="Delivery channel",
        ),
        "delivery_status": Column(
            str,
            Check.isin(DELIVERY_STATUS_VALUES),
            nullable=False,
            description="Delivery status",
        ),
        "acceptance_status": Column(
            str,
            Check.isin(ACCEPTANCE_STATUS_VALUES),
            nullable=False,
            description="Acceptance status",
        ),
        "outcome_tracked": Column(
            bool,
            nullable=False,
            description="Whether outcome is tracked",
        ),
        "outcome_value": Column(
            float,
            nullable=True,
            description="Outcome value if tracked",
        ),
        "trigger_reason": Column(
            str,
            nullable=False,
            description="Human-readable trigger reason",
        ),
        "causal_chain": Column(
            object,  # JSON/dict
            nullable=True,
            description="Causal chain evidence",
        ),
        "supporting_evidence": Column(
            object,  # JSON/dict
            nullable=True,
            description="Supporting evidence data",
        ),
        "recommended_action": Column(
            str,
            nullable=False,
            description="Recommended action text",
        ),
        "data_split": Column(
            str,
            Check.isin(DATA_SPLIT_VALUES),
            nullable=False,
            description="ML data split assignment",
        ),
    },
    strict=False,
    coerce=True,
    name="TriggerSchema",
    description="Schema for trigger records",
)


# =============================================================================
# BUSINESS METRICS SCHEMA
# =============================================================================

# Metric types for business metrics
METRIC_TYPES = ["trx", "nrx", "market_share", "conversion_rate", "hcp_engagement_score"]

BusinessMetricsSchema = DataFrameSchema(
    columns={
        "metric_id": Column(
            str,
            Check.str_matches(r"^metric_[a-f0-9]+$"),
            unique=True,
            description="Unique metric identifier",
        ),
        "metric_date": Column(
            str,
            Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"),
            nullable=False,
            description="Metric date (YYYY-MM-DD)",
        ),
        "metric_type": Column(
            str,
            Check.isin(METRIC_TYPES),
            nullable=False,
            description="Type of business metric",
        ),
        "metric_name": Column(
            str,
            nullable=False,
            description="Human-readable metric name",
        ),
        "brand": Column(
            str,
            Check.isin(BRAND_VALUES),
            nullable=False,
            description="Brand for this metric",
        ),
        "region": Column(
            str,
            Check.isin(REGION_VALUES),
            nullable=False,
            description="Geographic region",
        ),
        "value": Column(
            float,
            Check.ge(0),
            nullable=False,
            description="Metric value",
        ),
        "target": Column(
            float,
            Check.ge(0),
            nullable=False,
            description="Target value",
        ),
        "achievement_rate": Column(
            float,
            Check.in_range(0.0, 5.0),  # Allow up to 500% achievement
            nullable=False,
            description="Achievement rate (value/target)",
        ),
        "year_over_year_change": Column(
            float,
            Check.in_range(-1.0, 2.0),  # -100% to +200%
            nullable=False,
            description="Year-over-year change",
        ),
        "month_over_month_change": Column(
            float,
            Check.in_range(-0.5, 0.5),  # -50% to +50%
            nullable=False,
            description="Month-over-month change",
        ),
        "roi": Column(
            float,
            Check.ge(0),
            nullable=False,
            description="Return on investment",
        ),
        "statistical_significance": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Statistical significance (p-value)",
        ),
        "confidence_interval_lower": Column(
            float,
            nullable=False,
            description="Lower bound of confidence interval",
        ),
        "confidence_interval_upper": Column(
            float,
            nullable=False,
            description="Upper bound of confidence interval",
        ),
        "sample_size": Column(
            int,
            Check.ge(1),
            nullable=False,
            description="Sample size for this metric",
        ),
        "data_split": Column(
            str,
            Check.isin(DATA_SPLIT_VALUES),
            nullable=False,
            description="ML data split assignment",
        ),
    },
    strict=False,
    coerce=True,
    name="BusinessMetricsSchema",
    description="Schema for business metrics time-series records",
)


# =============================================================================
# FEATURE STORE SCHEMAS
# =============================================================================

# Feature value types from feature store schema
FEATURE_VALUE_TYPES = [
    "int64", "float64", "string", "bool", "timestamp",
    "array_int64", "array_float64", "array_string"
]

# Freshness status values
FRESHNESS_STATUS_VALUES = ["fresh", "stale", "expired"]

FeatureGroupsSchema = DataFrameSchema(
    columns={
        "id": Column(
            str,
            Check.str_matches(r"^[a-f0-9-]{36}$"),
            unique=True,
            description="UUID identifier for feature group",
        ),
        "name": Column(
            str,
            Check.str_length(min_value=1, max_value=255),
            nullable=False,
            description="Feature group name",
        ),
        "description": Column(
            str,
            nullable=True,
            description="Feature group description",
        ),
        "owner": Column(
            str,
            nullable=True,
            description="Owner team/person",
        ),
        "tags": Column(
            object,  # List
            nullable=True,
            description="Tags for categorization",
        ),
        "source_table": Column(
            str,
            nullable=True,
            description="Source table for features",
        ),
        "expected_update_frequency_hours": Column(
            int,
            Check.ge(1),
            nullable=False,
            description="Expected update frequency in hours",
        ),
        "max_age_hours": Column(
            int,
            Check.ge(1),
            nullable=False,
            description="Maximum age before stale in hours",
        ),
    },
    strict=False,
    coerce=True,
    name="FeatureGroupsSchema",
    description="Schema for feature group metadata",
)

FeaturesSchema = DataFrameSchema(
    columns={
        "id": Column(
            str,
            Check.str_matches(r"^[a-f0-9-]{36}$"),
            unique=True,
            description="UUID identifier for feature",
        ),
        "feature_group_id": Column(
            str,
            Check.str_matches(r"^[a-f0-9-]{36}$"),
            nullable=False,
            description="Parent feature group UUID",
        ),
        "name": Column(
            str,
            Check.str_length(min_value=1, max_value=255),
            nullable=False,
            description="Feature name",
        ),
        "description": Column(
            str,
            nullable=True,
            description="Feature description",
        ),
        "value_type": Column(
            str,
            Check.isin(FEATURE_VALUE_TYPES),
            nullable=False,
            description="Data type of feature value",
        ),
        "entity_keys": Column(
            object,  # List
            nullable=False,
            description="Entity keys for this feature",
        ),
        "owner": Column(
            str,
            nullable=True,
            description="Owner team/person",
        ),
        "tags": Column(
            object,  # List
            nullable=True,
            description="Tags for categorization",
        ),
        "drift_threshold": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            description="Threshold for drift detection",
        ),
    },
    strict=False,
    coerce=True,
    name="FeaturesSchema",
    description="Schema for feature metadata",
)

FeatureValuesSchema = DataFrameSchema(
    columns={
        "id": Column(
            str,
            Check.str_matches(r"^[a-f0-9-]{36}$"),
            unique=True,
            description="UUID identifier for feature value",
        ),
        "feature_id": Column(
            str,
            Check.str_matches(r"^[a-f0-9-]{36}$"),
            nullable=False,
            description="Parent feature UUID",
        ),
        "entity_values": Column(
            object,  # Dict/JSONB
            nullable=False,
            description="Entity identification values",
        ),
        "value": Column(
            object,  # JSONB
            nullable=False,
            description="Feature value as JSONB",
        ),
        "event_timestamp": Column(
            str,
            nullable=False,
            description="Event timestamp (ISO format)",
        ),
        "freshness_status": Column(
            str,
            Check.isin(FRESHNESS_STATUS_VALUES),
            nullable=False,
            description="Freshness status of the value",
        ),
    },
    strict=False,
    coerce=True,
    name="FeatureValuesSchema",
    description="Schema for feature value time-series records",
)


# =============================================================================
# SCHEMA REGISTRY
# =============================================================================

SCHEMA_REGISTRY: Dict[str, DataFrameSchema] = {
    "hcp_profiles": HCPProfileSchema,
    "patient_journeys": PatientJourneySchema,
    "treatment_events": TreatmentEventSchema,
    "ml_predictions": MLPredictionSchema,
    "triggers": TriggerSchema,
    "business_metrics": BusinessMetricsSchema,
    "feature_groups": FeatureGroupsSchema,
    "features": FeaturesSchema,
    "feature_values": FeatureValuesSchema,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    table_name: str,
    lazy: bool = True,
) -> Tuple[bool, Optional[pa.errors.SchemaErrors]]:
    """
    Validate a DataFrame against its schema.

    Args:
        df: DataFrame to validate
        table_name: Name of the table (must be in SCHEMA_REGISTRY)
        lazy: If True, collect all errors; if False, fail fast

    Returns:
        Tuple of (is_valid, errors)
    """
    if table_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown table: {table_name}. Available: {list(SCHEMA_REGISTRY.keys())}")

    schema = SCHEMA_REGISTRY[table_name]

    try:
        schema.validate(df, lazy=lazy)
        return True, None
    except pa.errors.SchemaErrors as e:
        return False, e


def validate_all_datasets(
    datasets: Dict[str, pd.DataFrame],
    lazy: bool = True,
) -> Dict[str, Tuple[bool, Optional[pa.errors.SchemaErrors]]]:
    """
    Validate all datasets against their schemas.

    Args:
        datasets: Dictionary of table_name -> DataFrame
        lazy: If True, collect all errors per table

    Returns:
        Dictionary of table_name -> (is_valid, errors)
    """
    results = {}

    for table_name, df in datasets.items():
        if table_name in SCHEMA_REGISTRY:
            results[table_name] = validate_dataframe(df, table_name, lazy=lazy)
        else:
            # Skip unknown tables
            results[table_name] = (True, None)

    return results


def get_validation_summary(
    results: Dict[str, Tuple[bool, Optional[pa.errors.SchemaErrors]]],
) -> str:
    """
    Generate a summary of validation results.

    Args:
        results: Output from validate_all_datasets

    Returns:
        Formatted summary string
    """
    lines = ["=" * 60, "PANDERA SCHEMA VALIDATION SUMMARY", "=" * 60]

    all_valid = True

    for table_name, (is_valid, errors) in results.items():
        status = "PASS" if is_valid else "FAIL"
        all_valid = all_valid and is_valid

        if is_valid:
            lines.append(f"  {table_name}: {status}")
        else:
            error_count = len(errors.failure_cases) if errors else 0
            lines.append(f"  {table_name}: {status} ({error_count} errors)")

            # Show first 3 errors
            if errors:
                for i, (_, case) in enumerate(errors.failure_cases.iterrows()):
                    if i >= 3:
                        lines.append(f"    ... and {error_count - 3} more errors")
                        break
                    lines.append(f"    - {case.get('column', 'unknown')}: {case.get('check', 'unknown')}")

    lines.append("-" * 60)
    overall = "ALL PASSED" if all_valid else "VALIDATION FAILED"
    lines.append(f"Overall: {overall}")
    lines.append("=" * 60)

    return "\n".join(lines)
