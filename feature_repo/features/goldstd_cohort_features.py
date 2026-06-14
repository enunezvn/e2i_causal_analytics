"""Gold-standard cohort RAW covariate feature view (#39).

The gold-standard cohort models (initiation / persistence / discontinuation ×
brand, ``*_goldstd_lr_v1``) are trained on a FeatureBuilder-encoded matrix whose
ONLY raw inputs are the 3 leakage-safe base covariates locked in
``src/mlops/gold_standard_eval/feature_builder.py`` (``KEEP_COLUMNS``):

    disease_severity, academic_hcp, geographic_region

This view serves those RAW covariates per ``patient_id`` so the real-time SHAP
explain route can fetch them from the Feast online store and hand them to the
BentoML service, which applies the bundled FeatureBuilder (raw -> 9 encoded
numeric features) before prediction. SHAP then runs over the encoded vector —
the audit-grade contract.

Why a NEW source/view (not the existing ``patient_journey_features``): that view
serves churn/adherence metrics (``days_on_therapy``, ``adherence_rate`` …) that
are POST-INDEX outcomes — the FeatureBuilder LEAKAGE_DENYLIST. The gold-standard
serving contract must expose ONLY the 3 pre-decision covariates, so it gets its
own view keyed on ``patient`` alone (the gold-standard models are brand-scoped at
the row level via the cohort, not via a composite patient_brand key).
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64, String

import sys

sys.path.append("..")
from entities import patient  # noqa: E402

# =============================================================================
# Data source — the 3 leakage-safe RAW covariates from patient_journeys
# =============================================================================
#
# Mirrors patient_journey_source's timestamp handling: ``event_date`` is the
# canonical event date (cast to TIMESTAMPTZ as event_timestamp) and ``created_at``
# is the created-timestamp column. Only the 3 KEEP_COLUMNS covariates are
# selected — no post-index/leakage columns enter this serving contract.
goldstd_cohort_source = PostgreSQLSource(
    name="goldstd_cohort_source",
    query="""
        SELECT
            patient_id::VARCHAR,
            event_date::TIMESTAMPTZ AS event_timestamp,
            created_at,
            disease_severity::DOUBLE PRECISION AS disease_severity,
            academic_hcp::BIGINT AS academic_hcp,
            geographic_region::VARCHAR AS geographic_region
        FROM patient_journeys
        WHERE event_date >= NOW() - INTERVAL '365 days'
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Gold-standard cohort RAW covariates (KEEP_COLUMNS) for SHAP serving.",
)


# =============================================================================
# Feature view — RAW covariates the gold-standard cohort models consume
# =============================================================================

goldstd_cohort_features_fv = FeatureView(
    name="goldstd_cohort_features",
    entities=[patient],
    ttl=timedelta(days=7),
    schema=[
        Field(
            name="disease_severity",
            dtype=Float64,
            description="Disease severity score (raw covariate, pre-index).",
        ),
        Field(
            name="academic_hcp",
            dtype=Int64,
            description="Whether the patient's HCP is academic (0/1, raw covariate).",
        ),
        Field(
            name="geographic_region",
            dtype=String,
            description="Patient geographic region (categorical raw covariate; "
            "one-hot-encoded by the bundled FeatureBuilder).",
        ),
    ],
    source=goldstd_cohort_source,
    online=True,
    tags={
        "use_case": "gold_standard_cohort_shap",
        "model_type": "binary_classification",
        "owner": "ml-foundation",
        "pii_category": "pseudonymized",
        "criticality": "high",
        "feature_set": "KEEP_COLUMNS",
    },
    description="RAW leakage-safe covariates (disease_severity, academic_hcp, "
    "geographic_region) for gold-standard cohort SHAP explanations (#39).",
)
