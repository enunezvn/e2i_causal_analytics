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

# Business metrics table - contains aggregated KPIs per HCP/territory.
# Reads from canonical business_metrics table; hcp_brand_id is a generated
# column (migration 033). Filter excludes the legacy per-brand+region
# aggregate rows (hcp_id IS NULL), so only the per-HCP rollup rows produced
# by the per-HCP business-metrics ETL (6B-infra-2a) flow into Feast.
business_metrics_source = PostgreSQLSource(
    name="business_metrics_source",
    query="""
        SELECT
            hcp_id::VARCHAR,
            territory_id::VARCHAR,
            brand_id::VARCHAR,
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
          AND brand_id IS NOT NULL AND brand_id <> ''
          AND territory_id IS NOT NULL AND territory_id <> ''
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
patient_journey_source = PostgreSQLSource(
    name="patient_journey_source",
    query="""
        SELECT
            patient_id::VARCHAR,
            brand_id::VARCHAR,
            patient_brand_id::VARCHAR,
            event_date AS event_timestamp,
            therapy_start_date,
            days_on_therapy,
            adherence_rate,
            refill_count,
            gap_days,
            is_churned,
            churn_risk_score,
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
