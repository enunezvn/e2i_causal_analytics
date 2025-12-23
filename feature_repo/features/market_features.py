"""Market dynamics feature views for ROI prediction and resource allocation.

Use Case: ROI Prediction
- Predict ROI for budget allocation decisions
- Features: market share trends, territory performance, investment levels
"""

from datetime import timedelta
from feast import Feature, FeatureView, Field
from feast.types import Float32, Float64, Int64, String

import sys
sys.path.append("..")
from entities import territory, brand, hcp_territory
from data_sources import business_metrics_source, territory_metrics_source


# =============================================================================
# Market Dynamics Feature View
# =============================================================================

market_dynamics_fv = FeatureView(
    name="market_dynamics_features",
    entities=[territory, brand],
    ttl=timedelta(days=7),
    schema=[
        # Volume metrics
        Field(name="trx_count", dtype=Int64, description="Territory TRx volume"),
        Field(name="nrx_count", dtype=Int64, description="Territory NRx volume"),
        Field(name="total_rx_count", dtype=Int64, description="Total prescriptions"),

        # Market position
        Field(name="market_share", dtype=Float32, description="Market share in territory"),

        # Performance indicators
        Field(name="conversion_rate", dtype=Float32, description="Territory conversion rate"),
        Field(name="engagement_score", dtype=Float32, description="Average engagement score"),
    ],
    source=business_metrics_source,
    online=True,
    tags={
        "use_case": "roi_prediction",
        "model_type": "regression",
        "owner": "ml-foundation",
        "criticality": "medium",
    },
    description="Market dynamics features for ROI prediction.",
)


# =============================================================================
# Territory Performance Feature View
# =============================================================================

territory_performance_fv = FeatureView(
    name="territory_performance_features",
    entities=[territory],
    ttl=timedelta(days=1),
    schema=[
        # Volume metrics
        Field(name="total_trx", dtype=Int64, description="Total TRx in territory"),
        Field(name="total_nrx", dtype=Int64, description="Total NRx in territory"),

        # Coverage metrics
        Field(name="active_hcp_count", dtype=Int64, description="Active HCPs in territory"),
        Field(name="covered_lives", dtype=Int64, description="Covered lives in territory"),

        # Potential and allocation
        Field(name="market_potential", dtype=Float32, description="Market potential score"),
        Field(name="resource_allocation_score", dtype=Float32, description="Current resource allocation"),
    ],
    source=territory_metrics_source,
    online=True,
    tags={
        "use_case": "roi_prediction",
        "feature_type": "dynamic",
        "freshness_requirement": "daily",
        "owner": "sales-ops",
    },
    description="Territory-level performance features for resource optimization.",
)
