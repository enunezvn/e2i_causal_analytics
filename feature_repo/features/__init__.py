"""Feature view definitions for E2I Causal Analytics.

This module exports all feature views organized by use case:
- HCP Conversion: Features for predicting HCP conversion likelihood
- Patient Journey: Features for churn prediction and adherence
- Trigger Effectiveness: Features for marketing campaign analysis
- Market Dynamics: Features for ROI prediction and resource allocation
"""

from .hcp_features import (
    hcp_conversion_fv,
    hcp_profile_fv,
    hcp_engagement_fv,
)
from .patient_features import (
    patient_journey_fv,
    patient_adherence_fv,
)
from .trigger_features import (
    trigger_effectiveness_fv,
    trigger_response_fv,
)
from .market_features import (
    market_dynamics_fv,
    territory_performance_fv,
)

__all__ = [
    # HCP features
    "hcp_conversion_fv",
    "hcp_profile_fv",
    "hcp_engagement_fv",
    # Patient features
    "patient_journey_fv",
    "patient_adherence_fv",
    # Trigger features
    "trigger_effectiveness_fv",
    "trigger_response_fv",
    # Market features
    "market_dynamics_fv",
    "territory_performance_fv",
]

# Feature view registry for programmatic access
FEATURE_VIEW_MAP = {
    "hcp_conversion": hcp_conversion_fv,
    "hcp_profile": hcp_profile_fv,
    "hcp_engagement": hcp_engagement_fv,
    "patient_journey": patient_journey_fv,
    "patient_adherence": patient_adherence_fv,
    "trigger_effectiveness": trigger_effectiveness_fv,
    "trigger_response": trigger_response_fv,
    "market_dynamics": market_dynamics_fv,
    "territory_performance": territory_performance_fv,
}


def get_feature_view(name: str):
    """Get a feature view by name."""
    if name not in FEATURE_VIEW_MAP:
        available = ", ".join(FEATURE_VIEW_MAP.keys())
        raise KeyError(f"Feature view '{name}' not found. Available: {available}")
    return FEATURE_VIEW_MAP[name]
