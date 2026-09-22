"""The real Feast FeatureView names and the offline source table each one reads FROM.

Cross-referenced to ``feature_repo/features/*.py`` (``source=...``) and
``feature_repo/data_sources.py`` (``FROM``). This is the ONE list; it is stdlib-only and
importable where ``import feast`` is not (the app/worker image, #307), so the Feast
tracking tables (#2207, ``src/tasks/feast_tracking.py``), the recency probe (#559,
``FeastClient._infer_source_table``) and the data-prep freshness gate
(``src/feature_store/feast_source_freshness.py``) cannot drift.
"""

from __future__ import annotations

from typing import Dict, List

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

# #559: per-source-table RAW timestamp column to MAX() for genuine recency. Keyed by the
# source table name (the unit that resolves in the statistics path). Feast's
# ``timestamp_field`` is ``event_timestamp`` for every source, but several of those are
# GENERATED columns derived from a raw base column (migration 033); we MAX() the raw base
# column so recency reflects the freshest underlying data point. Column + type verified
# live against the prod-equivalent Supabase. See FeastClient._infer_timestamp_column.
FEAST_SOURCE_TABLE_TIMESTAMP_COLUMNS: Dict[str, str] = {
    "business_metrics": "metric_date",  # DATE
    "patient_journeys": "journey_start_date",  # DATE
    "triggers": "trigger_timestamp",  # TIMESTAMPTZ (finer than derived trigger_date)
    "hcp_profiles": "updated_at",  # TIMESTAMPTZ (real, not generated)
    "territory_metrics": "metric_date",  # DATE
}


def feast_views_for_source_table(table: str | None) -> List[str]:
    """The Feast FeatureViews sourced FROM ``table`` (inverse of the map above), in
    declaration order. Empty for an unmapped / missing table — that table backs no
    Feast view, so Feast freshness is not a property of data loaded from it."""
    if not table:
        return []
    return [view for view, source in FEAST_FEATURE_VIEW_SOURCE_TABLES.items() if source == table]


__all__ = [
    "FEAST_FEATURE_VIEW_SOURCE_TABLES",
    "FEAST_SOURCE_TABLE_TIMESTAMP_COLUMNS",
    "feast_views_for_source_table",
]
