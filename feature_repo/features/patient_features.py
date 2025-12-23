"""Patient feature views for churn prediction and adherence analysis.

Use Case: Churn Prediction
- Predict likelihood of patient discontinuing therapy
- Features: adherence patterns, refill behavior, gap analysis
"""

from datetime import timedelta
from feast import Feature, FeatureView, Field
from feast.types import Bool, Float32, Float64, Int64, String, UnixTimestamp

import sys
sys.path.append("..")
from entities import patient, patient_brand
from data_sources import patient_journey_source


# =============================================================================
# Patient Journey Feature View
# =============================================================================

patient_journey_fv = FeatureView(
    name="patient_journey_features",
    entities=[patient, patient_brand],
    ttl=timedelta(days=7),
    schema=[
        # Therapy duration
        Field(name="days_on_therapy", dtype=Int64, description="Days since therapy start"),
        Field(name="therapy_start_date", dtype=UnixTimestamp, description="Therapy initiation date"),

        # Adherence metrics
        Field(name="adherence_rate", dtype=Float32, description="Adherence rate (0-1)"),
        Field(name="refill_count", dtype=Int64, description="Number of refills"),
        Field(name="gap_days", dtype=Int64, description="Cumulative gap days"),

        # Churn indicators
        Field(name="is_churned", dtype=Bool, description="Has patient churned"),
        Field(name="churn_risk_score", dtype=Float32, description="ML-derived churn risk (0-1)"),
    ],
    source=patient_journey_source,
    online=True,
    tags={
        "use_case": "churn_prediction",
        "model_type": "binary_classification",
        "owner": "ml-foundation",
        "pii_category": "pseudonymized",
        "criticality": "high",
    },
    description="Patient journey features for churn prediction.",
)


# =============================================================================
# Patient Adherence Feature View (Focused adherence metrics)
# =============================================================================

patient_adherence_fv = FeatureView(
    name="patient_adherence_features",
    entities=[patient, patient_brand],
    ttl=timedelta(days=1),  # Adherence needs frequent updates
    schema=[
        # Adherence metrics
        Field(name="adherence_rate", dtype=Float32, description="Current adherence rate"),
        Field(name="refill_count", dtype=Int64, description="Total refills to date"),
        Field(name="gap_days", dtype=Int64, description="Total gap days"),

        # Risk scoring
        Field(name="churn_risk_score", dtype=Float32, description="Current churn risk score"),
    ],
    source=patient_journey_source,
    online=True,
    tags={
        "use_case": "churn_prediction",
        "feature_type": "dynamic",
        "freshness_requirement": "daily",
        "owner": "ml-foundation",
        "pii_category": "pseudonymized",
    },
    description="Real-time patient adherence features for intervention targeting.",
)
