"""The real Feast FeatureView names and the offline source table each one reads FROM.

Cross-referenced to ``feature_repo/features/*.py`` (``source=...``) and
``feature_repo/data_sources.py`` (``FROM``). This is the ONE list; it is stdlib-only and
importable where ``import feast`` is not (the app/worker image, #307), so the Feast
tracking tables (#2207, ``src/tasks/feast_tracking.py``) and the recency probe (#559,
``FeastClient._infer_source_table``) cannot drift.
"""

from __future__ import annotations

from typing import Dict

FEAST_FEATURE_VIEW_SOURCE_TABLES: Dict[str, str] = {
    "hcp_conversion_features": "business_metrics",  # source=business_metrics_source
    "hcp_engagement_features": "business_metrics",  # source=business_metrics_source
    "hcp_profile_features": "hcp_profiles",  # source=hcp_profiles_source
    "patient_journey_features": "patient_journeys",  # source=patient_journey_source
    "patient_adherence_features": "patient_journeys",  # source=patient_journey_source
    "trigger_effectiveness_features": "triggers",  # source=triggers_source
    "trigger_response_features": "triggers",  # source=triggers_source
    "territory_performance_features": "territory_metrics",  # source=territory_metrics_source
    "market_dynamics_features": "business_metrics",  # source=business_metrics_source
}

__all__ = ["FEAST_FEATURE_VIEW_SOURCE_TABLES"]
