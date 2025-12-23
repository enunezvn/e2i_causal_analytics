"""Trigger feature views for marketing effectiveness analysis.

Use Case: Trigger Effectiveness
- Predict success likelihood of marketing triggers
- Features: response rates, conversion rates, channel performance
"""

from datetime import timedelta
from feast import Feature, FeatureView, Field
from feast.types import Bool, Float32, Float64, Int64, String

import sys
sys.path.append("..")
from entities import trigger, hcp, hcp_brand
from data_sources import triggers_source


# =============================================================================
# Trigger Effectiveness Feature View
# =============================================================================

trigger_effectiveness_fv = FeatureView(
    name="trigger_effectiveness_features",
    entities=[trigger, hcp, hcp_brand],
    ttl=timedelta(days=7),
    schema=[
        # Trigger metadata
        Field(name="trigger_type", dtype=String, description="Type of trigger (email, call, event)"),
        Field(name="channel", dtype=String, description="Delivery channel"),

        # Response metrics
        Field(name="is_responded", dtype=Bool, description="Did HCP respond to trigger"),
        Field(name="response_time_hours", dtype=Float32, description="Time to response in hours"),

        # Outcome metrics
        Field(name="conversion_flag", dtype=Bool, description="Did trigger lead to conversion"),
        Field(name="roi_estimate", dtype=Float32, description="Estimated ROI of trigger"),
    ],
    source=triggers_source,
    online=True,
    tags={
        "use_case": "trigger_effectiveness",
        "model_type": "binary_classification",
        "owner": "ml-foundation",
        "criticality": "medium",
    },
    description="Features for trigger effectiveness prediction.",
)


# =============================================================================
# Trigger Response Feature View (Real-time response tracking)
# =============================================================================

trigger_response_fv = FeatureView(
    name="trigger_response_features",
    entities=[trigger],
    ttl=timedelta(days=1),
    schema=[
        # Response metrics
        Field(name="is_responded", dtype=Bool, description="Has trigger been responded to"),
        Field(name="response_time_hours", dtype=Float32, description="Response time"),
        Field(name="conversion_flag", dtype=Bool, description="Conversion outcome"),
    ],
    source=triggers_source,
    online=True,
    tags={
        "use_case": "trigger_effectiveness",
        "feature_type": "dynamic",
        "freshness_requirement": "hourly",
        "owner": "ml-foundation",
    },
    description="Real-time trigger response tracking features.",
)
