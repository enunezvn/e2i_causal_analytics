"""HCP (Healthcare Provider) feature views for conversion prediction.

Use Case: HCP Conversion Prediction
- Predict likelihood of HCP converting to prescriber
- Features: engagement history, prescribing patterns, profile attributes
"""

import sys
from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Int64, String

sys.path.append("..")
from data_sources import (
    business_metrics_source,
    hcp_profiles_source,
)
from entities import hcp, hcp_brand

# =============================================================================
# HCP Conversion Feature View
# =============================================================================

hcp_conversion_fv = FeatureView(
    name="hcp_conversion_features",
    entities=[hcp, hcp_brand],
    ttl=timedelta(days=7),
    schema=[
        # Trigger funnel counts (per_hcp_rollup; renamed by migration 144 — not prescriptions)
        Field(
            name="triggers_delivered_count",
            dtype=Int64,
            description="Triggers delivered or viewed in period",
        ),
        Field(
            name="triggers_accepted_count",
            dtype=Int64,
            description="Triggers accepted or responded in period",
        ),
        Field(
            name="triggers_total_count",
            dtype=Int64,
            description="All triggers generated in period",
        ),
        # Market metrics
        Field(name="market_share", dtype=Float32, description="HCP's market share for brand"),
        Field(name="conversion_rate", dtype=Float32, description="Historical conversion rate"),
        # Engagement metrics
        Field(
            name="engagement_score", dtype=Float32, description="Overall engagement score (0-100)"
        ),
        Field(name="call_frequency", dtype=Float32, description="Rep call frequency (calls/month)"),
    ],
    source=business_metrics_source,
    online=True,
    tags={
        "use_case": "hcp_conversion",
        "model_type": "binary_classification",
        "owner": "ml-foundation",
        "criticality": "high",
    },
    description="Features for HCP conversion prediction model.",
)


# =============================================================================
# HCP Profile Feature View (Static/Semi-static attributes)
# =============================================================================

hcp_profile_fv = FeatureView(
    name="hcp_profile_features",
    entities=[hcp],
    ttl=timedelta(days=30),  # Profiles update less frequently
    schema=[
        # Demographics
        Field(name="specialty", dtype=String, description="Medical specialty"),
        Field(name="practice_type", dtype=String, description="Solo/Group/Hospital"),
        Field(name="years_of_practice", dtype=Int64, description="Years in practice"),
        # Segmentation
        Field(
            name="patient_volume_tier", dtype=String, description="High/Medium/Low patient volume"
        ),
        Field(name="digital_engagement_tier", dtype=String, description="Digital engagement level"),
        Field(name="prescribing_tier", dtype=String, description="Prescribing volume tier"),
    ],
    source=hcp_profiles_source,
    online=True,
    tags={
        "use_case": "hcp_conversion",
        "feature_type": "static",
        "owner": "ml-foundation",
    },
    description="Static HCP profile attributes for segmentation.",
)


# =============================================================================
# HCP Engagement Feature View (Time-varying engagement)
# =============================================================================

hcp_engagement_fv = FeatureView(
    name="hcp_engagement_features",
    entities=[hcp, hcp_brand],
    ttl=timedelta(days=1),  # Engagement features need freshness
    schema=[
        # Engagement metrics
        Field(name="engagement_score", dtype=Float32, description="Current engagement score"),
        Field(name="call_frequency", dtype=Float32, description="Recent call frequency"),
        Field(name="conversion_rate", dtype=Float32, description="Recent conversion rate"),
    ],
    source=business_metrics_source,
    online=True,
    tags={
        "use_case": "hcp_conversion",
        "feature_type": "dynamic",
        "freshness_requirement": "daily",
        "owner": "ml-foundation",
    },
    description="Time-varying HCP engagement features for real-time scoring.",
)
