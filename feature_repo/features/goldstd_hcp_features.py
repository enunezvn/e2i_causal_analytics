"""Gold-standard HCP-adoption RAW covariate feature view (#39 multi-model).

The per-brand HCP-adoption models (``hcp_adoption_{brand}_goldstd_lr_v1``) are
HCP-grain: their leakage-safe predictive covariates live on ``hcp_profiles``
(the SSOT for HCP attributes) and are JOIN-embedded by the gold-standard
FeatureBuilder HCP path. This view serves those 5 RAW covariates per ``hcp_id``
so the real-time SHAP explain route can fetch them and hand them to the
multi-model BentoML service, which applies the bundled (HCP) FeatureBuilder
(5 raw -> 19 encoded numeric features) before prediction. SHAP then runs over
the encoded vector — the audit-grade contract.

Distinct from the patient ``goldstd_cohort_features`` view: different ENTITY
(``hcp`` vs ``patient``) and a different (HCP) covariate set. ``specialty`` and
``geographic_region`` are categorical (one-hot at serve time); the other three
are numeric.
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64, String

import sys

sys.path.append("..")
from entities import hcp  # noqa: E402

# =============================================================================
# Data source — the 5 leakage-safe RAW HCP covariates from hcp_profiles
# =============================================================================
#
# hcp_profiles is the SSOT for HCP attributes. ``updated_at`` is the event
# timestamp; ``created_at`` is the created-timestamp column. Only the 5 covariate
# columns are selected — no post-decision/leakage columns enter this serving
# contract.
goldstd_hcp_source = PostgreSQLSource(
    name="goldstd_hcp_source",
    query="""
        SELECT
            hcp_id::VARCHAR,
            updated_at AS event_timestamp,
            created_at,
            peer_influence_score::DOUBLE PRECISION AS peer_influence_score,
            influence_network_size::BIGINT AS influence_network_size,
            years_experience::BIGINT AS years_experience,
            specialty::VARCHAR AS specialty,
            geographic_region::VARCHAR AS geographic_region
        FROM hcp_profiles
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Gold-standard HCP-adoption RAW covariates (hcp_profiles) for SHAP serving.",
)


# =============================================================================
# Feature view — RAW covariates the HCP-adoption models consume
# =============================================================================

goldstd_hcp_features_fv = FeatureView(
    name="goldstd_hcp_features",
    entities=[hcp],
    ttl=timedelta(days=30),
    schema=[
        Field(
            name="peer_influence_score",
            dtype=Float64,
            description="Peer influence score (numeric raw covariate).",
        ),
        Field(
            name="influence_network_size",
            dtype=Int64,
            description="Influence network size (numeric raw covariate).",
        ),
        Field(
            name="years_experience",
            dtype=Int64,
            description="Years of practice experience (numeric raw covariate).",
        ),
        Field(
            name="specialty",
            dtype=String,
            description="HCP specialty (categorical raw covariate; one-hot-encoded "
            "by the bundled FeatureBuilder).",
        ),
        Field(
            name="geographic_region",
            dtype=String,
            description="HCP geographic region (categorical raw covariate; "
            "one-hot-encoded by the bundled FeatureBuilder).",
        ),
    ],
    source=goldstd_hcp_source,
    online=True,
    tags={
        "use_case": "gold_standard_hcp_adoption_shap",
        "model_type": "binary_classification",
        "owner": "ml-foundation",
        "pii_category": "pseudonymized",
        "criticality": "high",
        "feature_set": "HCP_KEEP_COLUMNS",
    },
    description="RAW leakage-safe HCP covariates (peer_influence_score, "
    "influence_network_size, years_experience, specialty, geographic_region) for "
    "gold-standard HCP-adoption SHAP explanations (#39).",
)
