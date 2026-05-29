"""Data source definitions for E2I Causal Analytics Feature Store.

Data sources define where raw feature data comes from. All sources point to
Supabase PostgreSQL tables that contain the business data.

Tables are organized by domain:
- business_metrics: Core KPIs and operational metrics
- patient_journeys: Patient therapy data
- triggers: Marketing trigger events
- hcp_engagements: HCP interaction data
"""

from feast import Field, PushSource
from feast.data_source import RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import String, UnixTimestamp

# =============================================================================
# PostgreSQL Data Sources (Supabase)
# =============================================================================

# Business metrics table - per-HCP KPIs. Reads from canonical business_metrics;
# hcp_id / event_timestamp / hcp_brand_id and the metric columns are added by
# migration 033 (hcp_brand_id is a STORED generated column). The hcp_id IS NOT
# NULL filter excludes the legacy per-(brand, region) aggregate rows so only the
# per-HCP rollup rows produced by the 6B-infra-2a ETL flow into Feast.
#
# #556 drift fix: migration 033 made business_metrics per-HCP and never added
# territory_id or brand_id (the per-HCP ETL writes brand(enum)+region+hcp_id;
# hcp_brand_id is generated from hcp_id || enum_text(brand)). The prior
# SELECT/WHERE referenced territory_id + brand_id, which do not exist on the
# canonical table, so materialize() failed (UndefinedColumn). The HCP-keyed
# consumers (hcp_conversion_features, hcp_engagement_features) join on hcp_id +
# hcp_brand_id, both present. The territory+brand-keyed market_dynamics_features
# can no longer be served from this per-HCP source and is set online=False
# (see market_features.py).
business_metrics_source = PostgreSQLSource(
    name="business_metrics_source",
    query="""
        SELECT
            hcp_id::VARCHAR,
            hcp_brand_id::VARCHAR,
            event_timestamp,
            trx_count,
            nrx_count,
            total_rx_count,
            market_share,
            conversion_rate,
            engagement_score,
            call_frequency,
            created_at
        FROM business_metrics
        WHERE event_timestamp >= NOW() - INTERVAL '365 days'
          AND hcp_id IS NOT NULL AND hcp_id <> ''
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Business metrics from Supabase - TRx, NRx, market share, etc.",
)

# Patient journey table - therapy adherence and outcomes.
# Reads from canonical patient_journeys table; patient_brand_id, event_date,
# is_churned and brand_id are generated columns (migration 033).
# adherence_rate / refill_count / gap_days are populated by the per-patient
# adherence ETL (6B-infra-2b).
#
# #556 drift fix: the FeatureView fields therapy_start_date / days_on_therapy /
# churn_risk_score were aliases of bridging-view expressions that migration 033
# never promoted onto the canonical table. Map them to the real canonical
# columns: journey_start_date (therapy start), journey_duration_days (days on
# therapy), and risk_score (churn risk). COALESCE the two numeric fields so a
# NULL canonical value materializes as 0 rather than dropping the row.
patient_journey_source = PostgreSQLSource(
    name="patient_journey_source",
    query="""
        SELECT
            patient_id::VARCHAR,
            brand_id::VARCHAR,
            patient_brand_id::VARCHAR,
            event_date AS event_timestamp,
            journey_start_date AS therapy_start_date,
            COALESCE(journey_duration_days, 0) AS days_on_therapy,
            adherence_rate,
            refill_count,
            gap_days,
            is_churned,
            COALESCE(risk_score, 0) AS churn_risk_score,
            created_at
        FROM patient_journeys
        WHERE event_date >= NOW() - INTERVAL '365 days'
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Patient journey data for adherence and churn analysis.",
)

# Triggers table - marketing events and responses.
# Reads from canonical triggers table; hcp_brand_id, channel, is_responded,
# response_time_hours, conversion_flag and trigger_date are generated columns
# (migration 033). brand_id is NOT NULL after the migration 033 backfill, so
# the COALESCE-on-brand_id fallback the bridging view used is no longer
# needed.
triggers_source = PostgreSQLSource(
    name="triggers_source",
    query="""
        SELECT
            trigger_id::VARCHAR,
            hcp_id::VARCHAR,
            brand_id::VARCHAR,
            hcp_brand_id::VARCHAR,
            trigger_date AS event_timestamp,
            trigger_type,
            channel,
            is_responded,
            response_time_hours,
            conversion_flag,
            roi_estimate,
            created_at
        FROM triggers
        WHERE trigger_date >= NOW() - INTERVAL '365 days'
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Marketing trigger data for effectiveness analysis.",
)

# HCP profiles table - static and semi-static HCP attributes.
# Reads from canonical hcp_profiles table; uses real updated_at as
# event_timestamp (no 1h-backdate hack the bridging view applied).
# territory_id is backfilled to non-NULL ('UNASSIGNED' sentinel for any
# previously unmapped rows) by migration 033.
hcp_profiles_source = PostgreSQLSource(
    name="hcp_profiles_source",
    query="""
        SELECT
            hcp_id::VARCHAR,
            territory_id::VARCHAR,
            specialty,
            practice_type,
            patient_volume_tier,
            digital_engagement_tier,
            years_of_practice,
            prescribing_tier,
            updated_at AS event_timestamp,
            created_at
        FROM hcp_profiles
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="HCP profile attributes for targeting and segmentation.",
)

# Territory metrics table - geographic aggregations
territory_metrics_source = PostgreSQLSource(
    name="territory_metrics_source",
    query="""
        SELECT
            territory_id::VARCHAR,
            metric_date AS event_timestamp,
            total_trx,
            total_nrx,
            active_hcp_count,
            covered_lives,
            market_potential,
            resource_allocation_score,
            created_at
        FROM territory_metrics
        WHERE metric_date >= NOW() - INTERVAL '365 days'
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Territory-level metrics for resource optimization.",
)


# =============================================================================
# Push Sources (for real-time feature updates)
# =============================================================================

# Real-time HCP engagement events
hcp_engagement_push_source = PushSource(
    name="hcp_engagement_push_source",
    batch_source=business_metrics_source,
    description="Push source for real-time HCP engagement updates.",
)

# Real-time trigger response events
trigger_response_push_source = PushSource(
    name="trigger_response_push_source",
    batch_source=triggers_source,
    description="Push source for real-time trigger responses.",
)


# =============================================================================
# Request Sources (for on-demand feature computation)
# =============================================================================

# On-demand features computed at request time
hcp_request_source = RequestSource(
    name="hcp_request_source",
    schema=[
        Field(name="hcp_id", dtype=String),
        Field(name="brand_id", dtype=String),
        Field(name="request_timestamp", dtype=UnixTimestamp),
    ],
    description="Request source for on-demand HCP features.",
)


# =============================================================================
# Source Registry
# =============================================================================

ALL_SOURCES = [
    business_metrics_source,
    patient_journey_source,
    triggers_source,
    hcp_profiles_source,
    territory_metrics_source,
    hcp_engagement_push_source,
    trigger_response_push_source,
]

SOURCE_MAP = {s.name: s for s in ALL_SOURCES}


def get_source(name: str):
    """Get a data source by name."""
    if name not in SOURCE_MAP:
        available = ", ".join(SOURCE_MAP.keys())
        raise KeyError(f"Source '{name}' not found. Available: {available}")
    return SOURCE_MAP[name]
