"""Gold-standard cohort RAW covariate feature view (#39).

The gold-standard cohort models (initiation / persistence / discontinuation ×
brand, ``*_goldstd_lr_v1``) are trained on a FeatureBuilder-encoded matrix whose
raw inputs are the 7 leakage-safe ``_BASE7`` covariates in
``src/mlops/gold_standard_eval/cohort_spec.py`` after the T9/T11 DGP enrichment
(2026-06-21/22):

    disease_severity, academic_hcp, geographic_region,
    insurance_type, age_at_diagnosis, comorbidity_burden, prior_therapy_lines

(The 4 trailing prognostic drivers are drawn ⊥ treatment_arm so ATE/CATE are
preserved.) This view serves those RAW covariates per ``patient_id`` so the
real-time SHAP explain route can fetch them from the Feast online store and hand
them to the BentoML service, which applies the bundled FeatureBuilder (raw -> 19
encoded numeric features) before prediction. SHAP then runs over the encoded
vector — the audit-grade contract. Serving only the base 3 here feeds the
7-covariate bundle an incomplete vector (the #576 null-trap → 503).

The source window is widened to 2000 days because the synthetic cohort's
``event_date`` spans ~3 years (2023→2026); a tighter window would silently drop
older-dated patients from the online store, starving the cohort SHAP sample.

Why a NEW source/view (not the existing ``patient_journey_features``): that view
serves churn/adherence metrics (``days_on_therapy``, ``adherence_rate`` …) that
are POST-INDEX outcomes — the FeatureBuilder LEAKAGE_DENYLIST. The gold-standard
serving contract must expose ONLY the 3 pre-decision covariates, so it gets its
own view keyed on ``patient`` alone (the gold-standard models are brand-scoped at
the row level via the cohort, not via a composite patient_brand key).
"""

import sys
from datetime import timedelta

from feast import FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64, String

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
            geographic_region::VARCHAR AS geographic_region,
            insurance_type::VARCHAR AS insurance_type,
            age_at_diagnosis::BIGINT AS age_at_diagnosis,
            comorbidity_burden::BIGINT AS comorbidity_burden,
            prior_therapy_lines::BIGINT AS prior_therapy_lines,
            copay_support::BIGINT AS copay_support,
            psp_enrolled::BIGINT AS psp_enrolled,
            rep_detailing_high::BIGINT AS rep_detailing_high,
            sample_dropped::BIGINT AS sample_dropped
        FROM patient_journeys
        WHERE event_date >= NOW() - INTERVAL '2000 days'
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Gold-standard cohort RAW covariates (_BASE7) for SHAP serving.",
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
        # --- T9/T11 enrichment: 4 arm-independent prognostic drivers ---
        # Added to the patient DGP (2026-06-21/22) and to the _BASE7 cohort spec;
        # the enriched *_goldstd_lr_v1 models (feature_count=19) consume them, so
        # the online serving contract MUST expose them or the SHAP explain path
        # feeds an incomplete vector to the 7-covariate bundle (#576 null-trap).
        Field(
            name="insurance_type",
            dtype=String,
            description="Insurance access tier (categorical raw covariate; "
            "one-hot-encoded by the bundled FeatureBuilder).",
        ),
        Field(
            name="age_at_diagnosis",
            dtype=Int64,
            description="Patient age at diagnosis (numeric prognostic driver).",
        ),
        Field(
            name="comorbidity_burden",
            dtype=Int64,
            description="Comorbidity burden count (numeric prognostic driver).",
        ),
        Field(
            name="prior_therapy_lines",
            dtype=Int64,
            description="Number of prior therapy lines (numeric prognostic driver).",
        ),
        # --- COMM-ARMS Phase 1: the copay_support commercial arm ---
        # Consumed by the PERSISTENCE and DISCONTINUATION cohorts only. copay enters
        # the discontinuation logit, so it is real outcome signal; withholding it made
        # the persistence model blind to a driver it can legitimately observe (copay is
        # assigned pre-index and is NOT a leakage column). initiation does NOT consume
        # it — copay does not enter the treatment_initiated equation (verified: that
        # column is byte-identical pre/post copay), so serving it there would add a
        # pure-noise feature.
        #
        # NULL until the next synthetic reseed. That is safe rather than a trap: the
        # bundled FeatureBuilder median-imputes numerics and emits a `__isna` flag, so
        # an all-NULL column degrades to a constant feature, not a crash. It must still
        # be declared HERE, because a ref the view does not expose is the actual #576
        # null-trap.
        Field(
            name="copay_support",
            dtype=Int64,
            description="Copay-support commercial arm, 0/1 (persistence + "
            "discontinuation only; NULL until the next synthetic reseed).",
        ),
        # --- COMM-ARMS Phase 2: the psp_enrolled commercial arm ---
        # Same rationale as copay_support above: psp_enrolled enters the
        # discontinuation logit (Phase 2), so it is real outcome signal the
        # persistence/discontinuation models can legitimately observe (assigned
        # pre-index, not a leakage column). initiation does NOT consume it (psp is
        # not in the treatment_initiated equation). NULL until the next reseed;
        # declared here so the ref the persistence/disc views expose is not a
        # #576 null-trap.
        Field(
            name="psp_enrolled",
            dtype=Int64,
            description="Patient-support-program commercial arm, 0/1 (persistence + "
            "discontinuation only; NULL until the next synthetic reseed).",
        ),
        # --- COMM-ARMS Phase 3: rep_detailing_high + sample_dropped commercial arms ---
        # The MIRROR of copay/psp above: these two enter the INITIATION latent
        # (treatment_initiated via initiation_outcomes), so the initiation cohort consumes
        # them and persistence/discontinuation do NOT. Real pre-index outcome signal, not
        # leakage. NULL until the next synthetic reseed; declared here so the initiation
        # refs that request them are not a #576 null-trap.
        Field(
            name="rep_detailing_high",
            dtype=Int64,
            description="High-rep-detailing commercial arm, 0/1 (initiation only; "
            "NULL until the next synthetic reseed).",
        ),
        Field(
            name="sample_dropped",
            dtype=Int64,
            description="Sample-dropped commercial arm, 0/1 (initiation only; "
            "NULL until the next synthetic reseed).",
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
        "feature_set": "_BASE7",
    },
    description="RAW leakage-safe _BASE7 covariates (disease_severity, "
    "academic_hcp, geographic_region, insurance_type, age_at_diagnosis, "
    "comorbidity_burden, prior_therapy_lines) for gold-standard cohort SHAP "
    "explanations (#39 + T9/T11 enrichment).",
)
